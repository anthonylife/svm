/******************************************************
 * 基于SMO算法的支持向量机
 * 详情请见D:\下载资料\计算语言学\支持向量机\smo.pdf
 *****************************************************/
#include "smo.h"

//全局变量
int N=0;
int d=-1;
double C=0.05;
double tolerance=0.001;
double eps=0.001;
double two_sigma_squared=2;
double delta_b=0;

vector<double> alph;             //lagrange 乘因子
double b;
vector<double> w;                   //权系数向量w,仅用于线性的核函数

vector<double> error_cache;



//存储数据，只使用其中一个变量
vector<sparse_binary_vector> sparse_binary_points;
vector<sparse_vector> sparse_points;
vector<dense_vector> dense_points;

vector<int> target;   //训练数据的分类标签
bool is_sparse_data=false;
bool is_binary=false;
bool is_test_only=false;
bool is_linear_kernel=false;

int first_test_i=0;
int end_support_i=-1;
vector<double> precomputed_self_dot_product;

double (*dot_product_func)(int,int)=NULL;//计算两个样本之间的点积
double (*learned_func)(int)=NULL;//学习函数
double (*kernel_func)(int,int)=NULL;//核函数
 
int examineExample(int i1)
{
	double y1,alph1,E1,r1;
	y1=target[i1];
	alph1=alph[i1];
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];
	else
		E1=learned_func(i1)-y1;

	r1=y1*E1;
	if((r1-tolerance&&alph1<C)||(r1>tolerance&&alph1>0))//不满足KKT条件
	{
		//寻找第二个权值更新，并返回
		//寻找|E1-E2|最大的，18b
		{
			int k,i2;
			double tmax;
			for(i2=-1,tmax=0,k=0;k<end_support_i;k++)
				if(alph[k]>0&&alph[k]<C)
				{
					double E2,temp;					
					E2=error_cache[k];
					temp=fabs(E1-E2);
					if(temp>tmax)
					{
						tmax=temp;
						i2=k;
					}
				}
				if(i2>=0)
				{
					if(takeStep(i1,i2))
						return 1;
				}
		}
		//寻找边界样本19b
		{
			int k,k0;
			int i2;
            //采用随机数发生器的目的是随机设置遍历的起点位置，但是并不改变遍历所有变量的事实
			for(k0=(int)((rand()/RAND_MAX)*end_support_i),k=k0;k<end_support_i+k0;k++)
			{
				i2=k%end_support_i;
				if(alph[i2]>0&&alph[i2]<C)
				{
					if(takeStep(i1,i2))
						return 1;
				}
			}
		}
		//寻找整个样本19c
		{
			int k0,k,i2;
			for(k0=(int)((rand()/RAND_MAX)*end_support_i),k=k0;k<end_support_i+k0;k++)
			{
				i2=k%end_support_i;
				if(takeStep(i1,i2))
					return 1;
			}
		}

	}
	return 0;
}

int takeStep(int i1,int i2)
{
	int y1,y2,s;
	double alph1,alph2;//旧的权值
	double a1,a2;  //新的权值
	double E1,E2,L,H,k11,k22,k12,eta,lobj,hobj;

	if(i1==i2)
		return 0;
	//21a
	alph1=alph[i1];
	y1=target[i1];
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];
	else
		E1=learned_func(i1)-y1;
	alph2=alph[i2];
	y2=target[i2];
	if(alph2>0&&alph2<C)
		E2=error_cache[i2];
	else
		E2=learned_func(i2)-y2;


	s=y1*y2;
	//22a
	if(y1==y2)
	{
		double gamma=alph1+alph2;
		if(gamma>C)
		{
			L=gamma-C;
			H=C;
		}
		else
		{
			L=0;
			H=gamma;
		}
	}
	else
	{
		double gamma=alph1-alph2;
		if(gamma>0)
		{
			L=0;H=C-gamma;
		}
		else
		{
			L=-gamma;H=C;
		}
	}

	if(L==H)
		return 0;
	//22b
	k11=kernel_func(i1,i1);
	k12=kernel_func(i1,i2);
	k22=kernel_func(i2,i2);
	eta=2*k12-k11-k12;

	if(eta<0)
	{
		a2=alph2+y2*(E2-E1)/eta;
		if(a2<L)
			a2=L;
		else if(a2>H)
			a2=H;
	}
	else
	{
		//22d
        //eta等于0时的处理情况(eta<=0)
		double c1=eta/2;
		double c2=y2*(E1-E2)-eta*alph2;
		lobj=c1*L*L+c2*L;
		hobj=c1*H*H+c2*H;
	
		if(lobj>hobj+eps)
			a2=L;
		else if(lobj<hobj-eps)
			a2=H;
		else
			a2=alph2;
	}
    //a2的新旧值变化不大
	if(fabs(a2-alph2)<eps*(a2+alph2+eps))
		return 0;
	a1=alph1-s*(a2-alph2);
	if(a1<0)
	{
		a2+=s*a1;
		a1=0;
	}
	else if(a1>C)
	{
		double t=a1-C;
		a2+=s*t;
		a1=C;
	}
	//更新b 23a
	{
		double b1,b2,bnew;
		if(a1>0&&a1<C)
			bnew=b+E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12;
		else
		{
			if(a2>0&&a2<C)
				bnew=b+E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22;
			else
			{
				b1=b+E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12;
				b2=b+E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22;
				bnew=(b1+b2)/2;
			}
		}
		delta_b=bnew-b;
		b=bnew;
	}
	//如果使用线性的核函数，需要更新权向量 23c
	if(is_linear_kernel)
	{
		double t1=y1*(a1-alph1);
		double t2=y2*(a2-alph2);
		if(is_sparse_data&&is_binary)
		{
			int p1,num1,p2,num2;
			num1=(int)sparse_binary_points[i1].id.size();
			for(p1=0;p1<num1;p1++)
				w[sparse_binary_points[i1].id[p1]]+=t1;
			num2=(int)sparse_binary_points[i2].id.size();
			for(p2=0;p2<num2;p2++)
				w[sparse_binary_points[i2].id[p2]]+=t2;
		}
		else if(is_sparse_data&&!is_binary)
		{
			int p1,num1,p2,num2;
			num1=(int)sparse_points[i1].id.size();
			for(p1=0;p1<num1;p1++)
				w[sparse_points[i1].id[p1]]+=t1*sparse_points[i1].val[p1];
			num2=(int)sparse_points[i2].id.size();
			for(p2=0;p2<num2;p2++)
				w[sparse_points[i2].id[p2]]+=t2*sparse_points[i2].val[p2];
		}
		else
			for(int i=0;i<d;i++)
				w[i]+=dense_points[i1][i]*t1+dense_points[i2][i]*t2;
	}
	//更新错误率 24a
	{
		double t1=y1*(a1-alph1);
		double t2=y2*(a2-alph2);
		for(int i=0;i<end_support_i;i++)
			if(0<alph[i]&&alph[i]<C)
				error_cache[i]+=t1*kernel_func(i1,i)+t2*kernel_func(i2,i)-delta_b;
		error_cache[i1]=0;
		error_cache[i2]=0;
	}
	alph[i1]=a1;
	alph[i2]=a2;
	return 1;
}

double learned_func_linear_sparse_binary(int k)
{
	double s=0;
	for(int i=0;i<(int)sparse_binary_points[k].id.size();i++)
		s+=w[sparse_binary_points[k].id[i]];
	s-=b;
	return s;
}
double learned_func_linear_sparse_nobinary(int k)
{
	double s=0;
	for(int i=0;i<(int)sparse_points[k].id.size();i++)
	{
		int j=sparse_points[k].id[i];
		double v=sparse_points[k].val[i];
		s+=w[j]*v;
	}
	s-=b;
	return s;
}
double learned_func_linear_dense(int k)
{
	double s=0;
	for(int i=0;i<d;i++)
		s+=w[i]*dense_points[k][i];
	s-=b;
	return s;
}
double learned_func_nonlinear(int k)
{
	double s=0;
	for(int i=0;i<end_support_i;i++)
		if(alph[i]>0)
			s+=alph[i]*target[i]*kernel_func(i,k);
	s-=b;
	return s;
}
double dot_product_sparse_binary(int i1,int i2)
{
	int p1=0,p2=0,dot=0;
	int num1=(int)sparse_binary_points[i1].id.size();
	int num2=(int)sparse_binary_points[i2].id.size();
	while(p1<num1&&p2<num2)
	{
		int a1=(int)sparse_binary_points[i1].id[p1];
		int a2=(int)sparse_binary_points[i2].id[p2];
		if(a1==a2)
		{
			dot++;p1++;p2++;
		}
		else if(a1>a2)
			p2++;
		else
			p1++;
	}
	return (double)dot;
}
double dot_product_sparse_nonbinary(int i1,int i2)
{
	int p1=0,p2=0;
	double dot=0;
	int num1=(int)sparse_points[i1].id.size();
	int num2=(int)sparse_points[i2].id.size();
	while(p1<num1&&p2<num2)
	{
		int a1=sparse_points[i1].id[p1];
		int a2=sparse_points[i2].id[p2];
		if(a1==a2)
		{
			dot+=sparse_points[i1].val[p1]*sparse_points[i2].val[p2];
			p1++;
			p2++;
		}
		else if(a1>a2)
			p2++;
		else
			p1++;
	}
	return (double)dot;
}
double dot_product_dense(int i1,int i2)
{
	double dot=0;
	for(int i=0;i<d;i++)
		dot+=dense_points[i1][i]*dense_points[i2][i];
	return dot;
}
double rbf_kernel(int i1,int i2)
{
	double s=dot_product_func(i1,i2);
	s*=-2;
	s+=precomputed_self_dot_product[i1]+precomputed_self_dot_product[i2];
	return exp(-s/two_sigma_squared);
}
int read_data(istream& is)
{
	string s;
	int n_lines;
	for(n_lines=0;getline(is,s,'\n');n_lines++)
	{
		istrstream line(s.c_str());
		vector<double> v;
		double t;
		while(line>>t)
			v.push_back(t);
		target.push_back((int)v.back());
		v.pop_back();
		int n=(int)v.size();
		if(is_sparse_data&&is_binary)
		{
			sparse_binary_vector x;
			for(int i=0;i<n;i++)
			{
				if(v[i]<1||v[i]>d)
				{
#ifdef INFO
					cout<<"error:line"<<n_lines+1<<":attribute_index"<<int(v[i])<<"out of range."<<endl;
#endif
					return 0;
				}
				x.id.push_back(int(v[i])-1);
			}
		}
		else if (is_sparse_data&&!is_binary)
		{
			sparse_vector x;
			for(int i=0;i<n;i+=2)
			{
				if(v[i]<1||v[i]>d)
				{
#ifdef INFO
					cout<<"data file error:line"<<n_lines+1<<":attribute index "<<int(v[i])<<" out of range."<<endl;
#endif
					return 0;
				}
				x.id.push_back(int(v[i])-1);
				x.val.push_back(v[i+1]);
			}
			sparse_points.push_back(x);
		}
	else
	{
		if(v.size()!=d)
		{
#ifdef INFO
			cout<<"data file error:line "<<n_lines+1<<" has "<<(int)v.size()<<" attributes;should be d="<<d<<endl;
#endif
			return 0;
		}
		dense_points.push_back(v);
	}
	}
	return n_lines;
}
void write_svm(ostream& os)
{
	os<<d<<endl;
	os<<is_sparse_data<<endl;
	os<<is_binary<<endl;
	os<<is_linear_kernel<<endl;
	os<<b<<endl;
	if(is_linear_kernel)
	{
		for(int i=0;i<d;i++)
			os<<w[i]<<endl;
	}
	else
	{
		os<<two_sigma_squared<<endl;
		int n_support_vectors=0;
		for(int i=0;i<end_support_i;i++)
			if(alph[i]>0)
				n_support_vectors++;
		os<<n_support_vectors<<endl;
		for(int i=0;i<end_support_i;i++)
			if(alph[i]>0)
				os<<alph[i]<<endl;
		for(int i=0;i<end_support_i;i++)
			if(alph[i]>0)
			{
				if(is_sparse_data&&is_binary)
				{
					for(int j=0;j<(int)sparse_binary_points[i].id.size();j++)
						os<<(sparse_binary_points[i].id[j]+1)<<' ';
				}
				else if(is_sparse_data&&!is_binary)
				{
					for(int j=0;j<(int)sparse_points[i].id.size();j++)
						os<<(sparse_points[i].id[j]+1)<<' '<<sparse_points[i].val[j]<<' ';
				}
				else
				{
					for(int j=0;j<d;j++)
						os<<dense_points[i][j]<<' ';
				}
				os<<target[i];
				os<<endl;
			}
	}
}
int read_svm(istream& is)
{
	is>>d;
	is>>is_sparse_data;
	is>>is_binary;
	is>>is_linear_kernel;
	is>>b;
	if(is_linear_kernel)
	{
		w.resize(d);
		for(int i=0;i<d;i++)
			is>>w[i];
	}
	else
	{
		is>>two_sigma_squared;
		int n_support_vectors;
		is>>n_support_vectors;
		alph.resize(n_support_vectors,0);
		for(int i=0;i<n_support_vectors;i++)
			is>>alph[i];
		string dummy_line_to_skip_newline;
		getline(is,dummy_line_to_skip_newline,'\n');
		return read_data(is);
	}
	return 0;
}
double error_rate()
{
	int n_total=0;
	int n_error=0;
	for(int i=first_test_i;i<N;i++)
	{
		if(learned_func(i)>0!=target[i]>0)
			n_error++;
		n_total++;
	}
	return double(n_error)/double(n_total);
}
int smo(string data_file_name,string svm_file_name)
{
	//31a
	string output_file_name;
	int numChanged;
	int examineAll;
	//获得参数29d
	N=0;//训练样本的总数
	d=2;//样本空间的维数
	C=0.01;//惩罚因子
	tolerance=0.001;//满足KKT条件的容忍度
	eps=0.001;//控制变量更新时差值需要满足的最小值
	two_sigma_squared=2;//径向基核函数的参数
	data_file_name="svm.data";//数据文件
	svm_file_name="svm.model";//模型文件
	output_file_name="svm.output";//输出文件
	is_linear_kernel=false;//是否是线性的核函数
	is_sparse_data=false;//是否是稀疏数据
	is_binary=false;//是否是二进制数据
	is_test_only=false;
	
	//读入数据31c
	{
		int n;
		if(is_test_only)
		{
			ifstream svm_file(svm_file_name.c_str());
			end_support_i=first_test_i=n=read_svm(svm_file);
			N+=n;
		}
		if(N>0)
		{
			target.reserve(N);
			if(is_sparse_data&&is_binary)
				sparse_binary_points.reserve(N);
			else if(is_sparse_data&&!is_binary)
				sparse_points.reserve(N);
			else
				dense_points.reserve(N);
		}
		ifstream data_file(data_file_name.c_str());
		if(!data_file.is_open())
			return 1;
		n=read_data(data_file);
		if(n<=0)
			return 2;
		if(is_test_only)
		{
			N=first_test_i+n;
		}
		else
		{
			N=n;
			first_test_i=0;
			end_support_i=N;
		}
	}
	if(!is_test_only)
	{
		alph.resize(end_support_i,0.0);
		b=0;
		error_cache.resize(N);
		if(is_linear_kernel)
			w.resize(d,0.0);
	}

	//初始化学习函数,点积和核函数 26a
	if(is_linear_kernel&&is_sparse_data&&is_binary)
		learned_func=learned_func_linear_sparse_binary;
	if(is_linear_kernel&&is_sparse_data&&!is_binary)
		learned_func=learned_func_linear_sparse_nobinary;
	if(is_linear_kernel&&!is_sparse_data)
		learned_func=learned_func_linear_dense;
	if(!is_linear_kernel)
		learned_func=learned_func_nonlinear;
	if(is_sparse_data&&is_binary)
		dot_product_func=dot_product_sparse_binary;
	if(is_sparse_data&&!is_binary)
		dot_product_func=dot_product_sparse_nonbinary;
	if(!is_sparse_data)
		dot_product_func=dot_product_dense;
	if(is_linear_kernel)
		kernel_func=dot_product_func;
	if(!is_linear_kernel)
		kernel_func=rbf_kernel;
	if(!is_linear_kernel)
	{
		precomputed_self_dot_product.resize(N);
		for(int i=0;i<N;i++)
			precomputed_self_dot_product[i]=dot_product_func(i,i);
	}

	if(!is_test_only)
	{
		numChanged=0;
		examineAll=1;
		while(numChanged>0||examineAll)
		{
			numChanged=0;
			if(examineAll)
			{
				for(int k=0;k<N;k++)
					numChanged+=examineExample(k);
			}
			else
			{
                //保证先外层循环先遍历0<alpha<C的样本点，即间隔边界上的支持向量点
				for(int k=0;k<N;k++)
					if(alph[k]!=0&&alph[k]!=C)
						numChanged+=examineExample(k);
			}
			if(examineAll==1)
				examineAll=0;
			else if(numChanged==0)
				examineAll=1;
			//诊断信息36d
		}
		//输出模型参数36a
		{
			if((!is_test_only)&&(!svm_file_name.empty()))
			{
				ofstream svm_file(svm_file_name.c_str());
				write_svm(svm_file);
			}
		}
#ifdef INFO
		cout<<"threshold="<<b<<endl;
#endif
	}
#ifdef INFO
	cout<<"训练完毕，错误率为: "<<error_rate()<<endl;
#endif
	//输出分类36c
	return 0;
}
int load_svm(string svm_file_name)
{
	int n;
	//初始化学习函数和核函数

	N=0;
	d=2;//样本空间的维数
	C=0.01;//惩罚因子
	tolerance=0.001;//满足KKT条件的容忍度
	eps=0.001;
	ifstream svm_file(svm_file_name.c_str());
	if(!svm_file.is_open())
		return 1;
	end_support_i=first_test_i=n=read_svm(svm_file);
	if(n<=0)
		return 2;
	N+=n;
	return 0;
}
double predict_func(const vector<double>& vx)
{
	double s=0;
	for(int i=0;i<end_support_i;i++)
		if(alph[i]>0)
			s+=alph[i]*target[i]*kernel(dense_points[i],vx);
	s-=b;
	return s;
}
double kernel(const vector<double>& vx1,const vector<double>& vx2)
{
	if(vx1.size()!=vx2.size())
		return 0;
	double dot=0;
	for(int j=0;j<d;j++)
		dot+=(vx1[j]-vx2[j])*(vx1[j]-vx2[j]);
	dot/=2;
	return -dot;
} ...
