select * from df_orders

--top 10 highest revenue generating products

select product_id , sum(sales_price)
from df_orders
group by 1
order by 2 desc
limit 10

-- find top 5 highest selling products in each region


select * from df_orders

select region, product_id, total_sales
from (select region, product_id , sum(sales_price) as total_sales, 
dense_rank() over (partition by region order by sum(sales_price) desc) as rnk
from df_orders
group by 1,2)
where rnk <= 5

--find month over month growth comparison for 2022 and 2023 sales.

with s_22 as (select extract (month from order_date) as month, sum(sales_price) as sales_22
from df_orders 
where extract(year from order_date) = 2022
group by 1)
, s_23 as (select extract (month from order_date) as month, sum(sales_price) as sales_23
from df_orders
where extract(year from order_date) = 2023
group by 1)

select s.month, s.sales_22, t.sales_23 
from s_22 s
join s_23 t on s.month = t.month


--for each category which month had highest sales
select category , month_name
from (
	select TO_CHAR(order_date, 'YYYY-Mon') AS month_name, category , 
	sum(sales_price) as total_sales , 
	row_number() over (partition by category order by sum(sales_price) desc) as rnk
from df_orders
group by 1,2)
where rnk = 1


--which sub category has highest 
--growth by profit in 2023 compare to 2022



with s_22 as (select sub_category, sum(sales_price) as sales_22
from df_orders 
where extract(year from order_date) = 2022
group by 1)
, s_23 as (select sub_category, sum(sales_price) as sales_23
from df_orders
where extract(year from order_date) = 2023
group by 1)

select s.sub_category, s.sales_22, t.sales_23,
ROUND(
    (100.00 * (t.sales_23::NUMERIC / NULLIF(s.sales_22::NUMERIC, 0))),
    2
) AS growth_percentage

from s_22 s
join s_23 t on s.sub_category = t.sub_category




